void readline_show_prompt(ReadLineState *rs)

{

    monitor_printf(rs->mon, "%s", rs->prompt);

    monitor_flush(rs->mon);

    rs->last_cmd_buf_index = 0;

    rs->last_cmd_buf_size = 0;

    rs->esc_state = IS_NORM;

}
