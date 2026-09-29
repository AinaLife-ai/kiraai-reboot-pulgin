# 测试

本目录的测试**跑在真实的 KiraAI 框架上**（直接 import `core.*`），只替换掉需要跑起一个 bot 才有的东西（插件上下文、LLM 客户端、消息发送器、事件总线）。因此插件使用的是与生产环境同一份 `SessionManager`、同一批事件对象、同一个 schema 解析器。

## 运行

```bash
KIRA_CORE_PATH=/path/to/KiraAI python3 tests/run_tests.py
```

- `KIRA_CORE_PATH` 指向包含 `core/` 包的 KiraAI 检出目录（不设的话会尝试几个常见路径）。
- 每个用例都在独立的临时沙箱、独立的 `SessionManager` 与插件实例中运行。
- 常用参数：`--module test_summary`、`--filter async`。

## 反向验证（mutation check）

```bash
KIRA_CORE_PATH=/path/to/KiraAI python3 tests/reverse_check.py
```

这个脚本把 2.0 的**每一项关键设计决策逐个改回错误实现**，并断言对应测试确实变红。全绿说明测试真的在守护它声称守护的东西，而不是恒真断言。当前覆盖 15 项：

| 变异 | 必须变红的测试 |
|---|---|
| 清空方式默认改回 `delete_session` | `test_reset_keeps_title_description_timestamp_and_capabilities` |
| 摘要按「消息」而非「chunk」写回记忆 | `test_sync_summary_is_written_to_the_memory_head` |
| 摘要标记不再与 ADS 一致 | `test_summary_marker_is_byte_identical_to_ads` |
| 摘要重开关键词与 ADS 撞车 | `test_default_keywords_do_not_collide` |
| 去掉 LLM 工具门的每请求过滤 | `test_tool_hidden_from_the_model_when_disabled` |
| async 写回中间插入 await | `test_async_summary_does_not_lose_a_turn_written_meanwhile` |
| 摘要开关默认打开 | `test_reboot_keeps_summary_off_when_the_switch_is_off` |
| 白名单为空时放行 | `test_permission_on_with_empty_whitelist_denies_everyone` |
| 不再丢弃待处理缓冲 | `test_pending_buffer_is_dropped_before_clearing` |
| 去掉旧配置迁移 | `test_legacy_command_prefix_is_migrated` |
| 缓冲清空提前到摘要调用之前 | `test_buffer_is_dropped_at_write_time_not_before_the_summary_call` |
| 去掉 store 容量上限 | `test_store_is_capped_and_evicts_the_oldest` |
| 短 id 反向包含匹配 | `test_short_plugin_ids_do_not_produce_bogus_conflict_reports` |
| 去掉 notice 拦截 | `test_notice_messages_cannot_trigger_a_reset` |
| 去掉某个配置项 hint | `test_every_config_item_has_a_hint` |

## 用例分布

| 文件 | 用例 | 覆盖 |
|---|---|---|
| `test_reset.py` | 17 | 清空语义（元信息保留 / 删除模式对照 / 反向验证旧实现）、空历史、**缓冲丢弃时机**、并发锁、冷却、事件订阅、生命周期 |
| `test_summary.py` | 22 | sync/async、累积合并、自压缩与截断、失败降级、**async 零 await 竞态**、摘要保活桥、**重复重开的幂等性**、**store 容量上限** |
| `test_prompt_layout.py` | 6 | **提示词布局与前缀缓存**：摘要的位置、动态段重定位、跨轮前缀稳定性、内存桥布局一致性 |
| `test_commands.py` | 15 | 整句匹配、拦截保证（discard+stop）、**notice 不可触发**、权限（fail-closed）、文案模板 |
| `test_tool_gate.py` | 13 | 工具注册名、双层门、权限、冷却、异常路径 |
| `test_ads_compat.py` | 14 | 摘要标记契约、ADS 探测、指令让位、**短 id 假阳性**、告警 |
| `test_config_migration.py` | 14 | schema 解析（真解析器）、默认值设计、1.0 扁平配置迁移、**数据目录缺失时降级** |
| `test_version_bump.py` | 11 | manifest / schema / 更新日志一致性守卫、**死配置检查**、**README 漂移检查**、hint 完整性 |

## 辅助脚本

- `harness.py` — 沙箱、假件与夹具。
- `reverse_check.py` — 反向验证。
