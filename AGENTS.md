# OpenFang Source

这个目录是 OpenFang 子仓源码。这里不是一次性导入的第三方代码，而是需要长期维护的 fork。

作用：

- 承载 OpenFang 内核、API、runtime、channels、skills 等源码
- 作为 OpenFang 对上游 `openfang` 的长期 fork 维护点
- 承接那些确实必须落到上游框架层的改动

远端约定：

- `origin`：`git@github.com:LJJ/openfang.git`
- `upstream`：`https://github.com/RightNow-AI/openfang.git`

分支与同步规则：

- 默认把 `main` 当成长期维护主线，父仓 submodule 只引用这里已经存在的 commit
- 不要对已经推到 `origin/main`、且可能被父仓引用的历史做 `rebase` 或 `force-push`
- 日常开发先从 `main` 切功能分支，完成后再合回 `main`
- 同步上游时，优先新开 `sync/upstream-YYYYMMDD` 之类的分支，把 `upstream/main` merge 进来，解决冲突并验证后再合回 `main`
- 同步上游的主线优先用 `merge`，不要反复改写共享历史

改动边界：

- 能留在父仓 `/home/ljj/openfang` 或 `.openfang/` 配置层的差异，不优先下沉到这个子仓
- 只有确实需要改 OpenFang 通用 runtime / kernel / API / channel 能力时，才在这里改
- 如果某个改动具备上游通用价值，优先考虑整理成可回馈 upstream 的形式
- 纯 OpenFang 业务语义、宋玉人设、提示词策略、运行态编排，优先放父仓，不要无谓扩大 fork 与 upstream 的差异面

父仓协作规则：

- 改完这个子仓并提交后，父仓需要单独更新 submodule 指针
- 父仓 bump 子仓指针时，尽量单独成 commit，方便回溯“哪次升级了子仓”
- 在子仓 commit 没有推到 `origin` 前，不要 push 父仓里引用该 commit 的 submodule 指针

忽略与产物规则：

- 构建产物、测试产物、临时目录不要进版本库
- 当前已忽略：`.cargo-target/`、`target-codex/`、`target-codex-tests/`
- 若后续新增本地缓存或调试目录，要同步更新这里的 `.gitignore`

使用方式：

- 需要改内核、API、channel、runtime、bundled agent、bundled skill 时，在这里做
- 构建与校验优先用 debug 流程，不默认追求 release
- 改动完先在子仓内自检，再回父仓更新 submodule 指针和相关文档

详细文档：

- `../../docs/source-tree.md`
- `../../docs/architecture.md`
- `../../docs/session-projection.md`
- `../../docs/debug-runtime-mode.md`
