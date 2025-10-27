static void readline_update(ReadLineState *rs)

{

    int i, delta, len;



    if (rs->cmd_buf_size != rs->last_cmd_buf_size ||

        memcmp(rs->cmd_buf, rs->last_cmd_buf, rs->cmd_buf_size) != 0) {

        for(i = 0; i < rs->last_cmd_buf_index; i++) {

            monitor_printf(rs->mon, "\033[D");

        }

        rs->cmd_buf[rs->cmd_buf_size] = '\0';

        if (rs->read_password) {

            len = strlen(rs->cmd_buf);

            for(i = 0; i < len; i++)

                monitor_printf(rs->mon, "*");

        } else {

            monitor_printf(rs->mon, "%s", rs->cmd_buf);

        }

        monitor_printf(rs->mon, "\033[K");

        memcpy(rs->last_cmd_buf, rs->cmd_buf, rs->cmd_buf_size);

        rs->last_cmd_buf_size = rs->cmd_buf_size;

        rs->last_cmd_buf_index = rs->cmd_buf_size;

    }

    if (rs->cmd_buf_index != rs->last_cmd_buf_index) {

        delta = rs->cmd_buf_index - rs->last_cmd_buf_index;

        if (delta > 0) {

            for(i = 0;i < delta; i++) {

                monitor_printf(rs->mon, "\033[C");

            }

        } else {

            delta = -delta;

            for(i = 0;i < delta; i++) {

                monitor_printf(rs->mon, "\033[D");

            }

        }

        rs->last_cmd_buf_index = rs->cmd_buf_index;

    }

    monitor_flush(rs->mon);

}
