static int write_payload_3270(EmulatedCcw3270Device *dev, uint8_t cmd)

{

    Terminal3270 *t = TERMINAL_3270(dev);

    int retval = 0;

    int count = ccw_dstream_avail(get_cds(t));



    assert(count <= (OUTPUT_BUFFER_SIZE - 3) / 2);



    if (!t->handshake_done) {

        if (!(t->outv[0] == IAC && t->outv[1] != IAC)) {

            /*

             * Before having finished 3270 negotiation,

             * sending outbound data except protocol options is prohibited.

             */

            return 0;

        }

    }

    if (!qemu_chr_fe_backend_connected(&t->chr)) {

        /* We just say we consumed all data if there's no backend. */

        return count;

    }

    t->outv[0] = cmd;

    ccw_dstream_read_buf(get_cds(t), &t->outv[1], count);

    t->out_len = count + 1;



    t->out_len = insert_IAC_escape_char(t->outv, t->out_len);

    t->outv[t->out_len++] = IAC;

    t->outv[t->out_len++] = IAC_EOR;



    retval = qemu_chr_fe_write_all(&t->chr, t->outv, t->out_len);

    return (retval <= 0) ? 0 : (retval - 3);

}
