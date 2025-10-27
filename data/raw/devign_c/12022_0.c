static void dump_ops(const uint16_t *opc_buf)

{

    const uint16_t *opc_ptr;

    int c;

    opc_ptr = opc_buf;

    for(;;) {

        c = *opc_ptr++;

        fprintf(logfile, "0x%04x: %s\n", opc_ptr - opc_buf - 1, op_str[c]);

        if (c == INDEX_op_end)

            break;

    }

}
