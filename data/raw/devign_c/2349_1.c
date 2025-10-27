static inline bool handler_is_async(const mon_cmd_t *cmd)

{

    return cmd->flags & MONITOR_CMD_ASYNC;

}
