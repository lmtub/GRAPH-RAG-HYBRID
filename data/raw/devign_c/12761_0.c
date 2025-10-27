void readline_handle_byte(ReadLineState *rs, int ch)

{

    switch(rs->esc_state) {

    case IS_NORM:

        switch(ch) {

        case 1:

            readline_bol(rs);

            break;

        case 4:

            readline_delete_char(rs);

            break;

        case 5:

            readline_eol(rs);

            break;

        case 9:

            readline_completion(rs);

            break;

        case 10:

        case 13:

            rs->cmd_buf[rs->cmd_buf_size] = '\0';

            if (!rs->read_password)

                readline_hist_add(rs, rs->cmd_buf);

            monitor_printf(rs->mon, "\n");

            rs->cmd_buf_index = 0;

            rs->cmd_buf_size = 0;

            rs->last_cmd_buf_index = 0;

            rs->last_cmd_buf_size = 0;

            rs->readline_func(rs->mon, rs->cmd_buf, rs->readline_opaque);

            break;

        case 23:

            /* ^W */

            readline_backword(rs);

            break;

        case 27:

            rs->esc_state = IS_ESC;

            break;

        case 127:

        case 8:

            readline_backspace(rs);

            break;

	case 155:

            rs->esc_state = IS_CSI;

	    break;

        default:

            if (ch >= 32) {

                readline_insert_char(rs, ch);

            }

            break;

        }

        break;

    case IS_ESC:

        if (ch == '[') {

            rs->esc_state = IS_CSI;

            rs->esc_param = 0;

        } else if (ch == 'O') {

            rs->esc_state = IS_SS3;

            rs->esc_param = 0;

        } else {

            rs->esc_state = IS_NORM;

        }

        break;

    case IS_CSI:

        switch(ch) {

	case 'A':

	case 'F':

	    readline_up_char(rs);

	    break;

	case 'B':

	case 'E':

	    readline_down_char(rs);

	    break;

        case 'D':

            readline_backward_char(rs);

            break;

        case 'C':

            readline_forward_char(rs);

            break;

        case '0' ... '9':

            rs->esc_param = rs->esc_param * 10 + (ch - '0');

            goto the_end;

        case '~':

            switch(rs->esc_param) {

            case 1:

                readline_bol(rs);

                break;

            case 3:

                readline_delete_char(rs);

                break;

            case 4:

                readline_eol(rs);

                break;

            }

            break;

        default:

            break;

        }

        rs->esc_state = IS_NORM;

    the_end:

        break;

    case IS_SS3:

        switch(ch) {

        case 'F':

            readline_eol(rs);

            break;

        case 'H':

            readline_bol(rs);

            break;

        }

        rs->esc_state = IS_NORM;

        break;

    }

    readline_update(rs);

}
