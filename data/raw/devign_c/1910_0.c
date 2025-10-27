static int mux_proc_byte(CharDriverState *chr, MuxDriver *d, int ch)

{

    if (d->term_got_escape) {

        d->term_got_escape = 0;

        if (ch == term_escape_char)

            goto send_char;

        switch(ch) {

        case '?':

        case 'h':

            mux_print_help(chr);

            break;

        case 'x':

            {

                 const char *term =  "QEMU: Terminated\n\r";

                 qemu_chr_fe_write(chr, (uint8_t *)term, strlen(term));

                 exit(0);

                 break;

            }

        case 's':

            blk_commit_all();

            break;

        case 'b':

            qemu_chr_be_event(chr, CHR_EVENT_BREAK);

            break;

        case 'c':

            /* Switch to the next registered device */

            mux_chr_send_event(d, d->focus, CHR_EVENT_MUX_OUT);

            d->focus++;

            if (d->focus >= d->mux_cnt)

                d->focus = 0;

            mux_chr_send_event(d, d->focus, CHR_EVENT_MUX_IN);

            break;

        case 't':

            d->timestamps = !d->timestamps;

            d->timestamps_start = -1;

            d->linestart = 0;

            break;

        }

    } else if (ch == term_escape_char) {

        d->term_got_escape = 1;

    } else {

    send_char:

        return 1;

    }

    return 0;

}
