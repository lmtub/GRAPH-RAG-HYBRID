static void term_handle_byte(int ch)

{

    switch(term_esc_state) {

    case IS_NORM:

        switch(ch) {

        case 1:

            term_bol();

            break;

        case 4:

            term_delete_char();

            break;

        case 5:

            term_eol();

            break;

        case 9:

            term_completion();

            break;

        case 10:

        case 13:

            term_cmd_buf[term_cmd_buf_size] = '\0';

	    term_hist_add(term_cmd_buf);

            term_printf("\n");

            term_handle_command(term_cmd_buf);

            term_show_prompt();

            break;

        case 27:

            term_esc_state = IS_ESC;

            break;

        case 127:

        case 8:

            term_backspace();

            break;

	case 155:

            term_esc_state = IS_CSI;

	    break;

        default:

            if (ch >= 32) {

                term_insert_char(ch);

            }

            break;

        }

        break;

    case IS_ESC:

        if (ch == '[') {

            term_esc_state = IS_CSI;

            term_esc_param = 0;

        } else {

            term_esc_state = IS_NORM;

        }

        break;

    case IS_CSI:

        switch(ch) {

	case 'A':

	case 'F':

	    term_up_char();

	    break;

	case 'B':

	case 'E':

	    term_down_char();

	    break;

        case 'D':

            term_backward_char();

            break;

        case 'C':

            term_forward_char();

            break;

        case '0' ... '9':

            term_esc_param = term_esc_param * 10 + (ch - '0');

            goto the_end;

        case '~':

            switch(term_esc_param) {

            case 1:

                term_bol();

                break;

            case 3:

                term_delete_char();

                break;

            case 4:

                term_eol();

                break;

            }

            break;

        default:

            break;

        }

        term_esc_state = IS_NORM;

    the_end:

        break;

    }

    term_update();

}
