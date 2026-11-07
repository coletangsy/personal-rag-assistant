import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import * as fs from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import vm from 'node:vm';

const workflow = fs.readFileSync(new URL('../workflows/delete-merged-branch.yml', import.meta.url), 'utf8');
const source = workflow.match(/node <<'NODE'\n([\s\S]*?)\n +NODE/)[1];
const repository = 'example/repository';

async function run({ merged = true, branch = 'codex/topic', headRepository = repository,
  protectedBranch = false, headSha = 'merged-head', currentSha = headSha, afterPushSha = currentSha,
  getStatus = 200, afterPushStatus = getStatus, openStatus = 200, baseStatus = openStatus,
  openPRs = [], basePRs = [],
  pushError = false, realGit = false, beforePush = () => {}, server = 'https://github.com' } = {}) {
  const directory = fs.mkdtempSync(join(tmpdir(), 'branch-cleanup-test-'));
  const event = { repository: { default_branch: 'main' }, pull_request: {
    merged, base: { ref: 'release' },
    head: { ref: branch, sha: headSha, repo: headRepository === null ? null : { full_name: headRepository } },
  } };
  const calls = [];
  const gitCalls = [];
  let attemptedPush = false;
  const process = { env: {
    GITHUB_EVENT_PATH: '/event.json', GITHUB_REPOSITORY: repository,
    GITHUB_API_URL: 'https://api.github.com', GITHUB_SERVER_URL: server,
    GH_TOKEN: 'test-token', RUNNER_TEMP: directory,
  }, exitCode: 0 };
  try {
    await vm.runInNewContext(source, {
      require(name) {
        if (name === 'node:fs') return { ...fs, readFileSync: () => JSON.stringify(event) };
        if (name === 'node:path') return { join };
        assert.equal(name, 'node:child_process');
        return { execFileSync(command, args, options) {
          gitCalls.push({ command, args, options });
          if (args[0] === 'push') {
            attemptedPush = true;
            beforePush();
            if (pushError) throw new Error('Git rejected the push');
          }
          return realGit ? execFileSync(command, args, options) : Buffer.from('');
        } };
      },
      process, AbortSignal, Buffer,
      console: { log() {}, error() {} },
      async fetch(url, options) {
        calls.push({ url, ...options });
        const open = url.includes('/pulls?');
        const base = url.includes('&base=');
        const status = open ? base ? baseStatus : openStatus : attemptedPush ? afterPushStatus : getStatus;
        const sha = attemptedPush ? afterPushSha : currentSha;
        return { status, ok: status >= 200 && status < 300,
          json: async () => open ? base ? basePRs : openPRs : { ...(protectedBranch === null ? {} : { protected: protectedBranch }), commit: { sha } } };
      },
    }, { timeout: 1000 });
    return { calls, gitCalls, exitCode: process.exitCode };
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

test('deletes the merged tip with an explicit SHA lease and argument-safe ref', async () => {
  const branch = 'codex/topic-$(echo_test)';
  const { calls, gitCalls, exitCode } = await run({ branch });
  assert.equal(exitCode, 0);
  assert(calls[0].url.endsWith(`/branches/${encodeURIComponent(branch)}`));
  assert.equal(calls[0].redirect, 'error');
  assert(calls[1].url.includes(`head=${encodeURIComponent(`example:${branch}`)}`));
  assert(calls[2].url.includes(`base=${encodeURIComponent(branch)}`));
  assert.deepEqual([...gitCalls[1].args], ['push', '--porcelain',
    `--force-with-lease=refs/heads/${branch}:merged-head`,
    `https://github.com/${repository}.git`, `:refs/heads/${branch}`]);
  assert.equal(gitCalls[1].options.env.GIT_CONFIG_KEY_0, 'http.https://github.com/.extraheader');
  assert.equal(gitCalls[1].options.env.GIT_CONFIG_VALUE_1, '');
});

for (const [name, options] of [
  ['an unmerged PR', { merged: false }],
  ['a fork branch', { headRepository: 'contributor/repository' }],
  ['a deleted head repository', { headRepository: null }],
  ['a missing merged head SHA', { headSha: null }],
  ['an empty branch name', { branch: '' }],
  ['the default branch', { branch: 'main' }],
  ['the base branch', { branch: 'release' }],
]) {
  test(`does not access GitHub for ${name}`, async () => {
    const { calls, gitCalls, exitCode } = await run(options);
    assert.equal(exitCode, 0);
    assert.deepEqual(calls, []);
    assert.deepEqual(gitCalls, []);
  });
}

for (const [name, options, requests] of [
  ['an already deleted branch', { getStatus: 404 }, 1],
  ['a protected branch', { protectedBranch: true }, 1],
  ['missing protection metadata', { protectedBranch: null }, 1],
  ['new commits after the merge', { currentSha: 'new-head' }, 1],
  ['a branch used by another open PR', { openPRs: [{ number: 2 }] }, 2],
  ['a branch used as an open PR base', { basePRs: [{ number: 3 }] }, 3],
]) {
  test(`does not delete ${name}`, async () => {
    const { calls, gitCalls, exitCode } = await run(options);
    assert.equal(exitCode, 0);
    assert.equal(calls.length, requests);
    assert.deepEqual(gitCalls, []);
  });
}

for (const [name, options] of [
  ['a forbidden branch read', { getStatus: 403 }],
  ['a forbidden open PR lookup', { openStatus: 403 }],
  ['a forbidden open PR base lookup', { baseStatus: 403 }],
  ['a rejected push with an unchanged tip', { pushError: true }],
  ['a failed post-push recheck', { pushError: true, afterPushStatus: 403 }],
]) {
  test(`fails for ${name}`, async () => {
    assert.equal((await run(options)).exitCode, 1);
  });
}

test('accepts a branch already deleted before a rejected push', async () => {
  assert.equal((await run({ pushError: true, afterPushStatus: 404 })).exitCode, 0);
});

function git(directory, args, input) {
  return execFileSync('git', ['-c', 'user.name=Workflow test', '-c', 'user.email=test@example.invalid', ...args],
    { cwd: directory, input, encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
}

for (const changedBeforePush of [false, true]) {
  test(`real Git ${changedBeforePush ? 'preserves a tip pushed after the API check' : 'deletes the unchanged merged tip'}`, async () => {
    const directory = fs.mkdtempSync(join(tmpdir(), 'branch-cleanup-remote-'));
    const server = join(directory, 'server');
    const remote = join(server, `${repository}.git`);
    try {
      fs.mkdirSync(dirname(remote), { recursive: true });
      git(directory, ['init', '--bare', '--quiet', remote]);
      const tree = git(remote, ['mktree'], '');
      const merged = git(remote, ['commit-tree', tree], 'Merged\n');
      const newer = git(remote, ['commit-tree', tree, '-p', merged], 'New after merge\n');
      git(remote, ['update-ref', 'refs/heads/codex/topic', merged]);
      const result = await run({ realGit: true, server: pathToFileURL(server).href,
        headSha: merged, afterPushSha: changedBeforePush ? newer : merged,
        beforePush: () => { if (changedBeforePush) git(remote, ['update-ref', 'refs/heads/codex/topic', newer]); },
      });
      assert.equal(result.exitCode, 0);
      assert(result.gitCalls.some(call => call.args[0] === 'push'));
      const remaining = git(remote, ['for-each-ref', '--format=%(objectname)', 'refs/heads/codex/topic']);
      assert.equal(remaining, changedBeforePush ? newer : '');
    } finally {
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });
}
