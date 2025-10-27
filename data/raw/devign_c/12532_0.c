static void mux_print_help(CharDriverState *chr)

{

    int i, j;

    char ebuf[15] = "Escape-Char";

    char cbuf[50] = "\n\r";



    if (term_escape_char > 0 && term_escape_char < 26) {

        snprintf(cbuf, sizeof(cbuf), "\n\r");

        snprintf(ebuf, sizeof(ebuf), "C-%c", term_escape_char - 1 + 'a');

    } else {

        snprintf(cbuf, sizeof(cbuf),

                 "\n\rEscape-Char set to Ascii: 0x%02x\n\r\n\r",

                 term_escape_char);

    }

    qemu_chr_fe_write(chr, (uint8_t *)cbuf, strlen(cbuf));

    for (i = 0; mux_help[i] != NULL; i++) {

        for (j=0; mux_help[i][j] != '\0'; j++) {

            if (mux_help[i][j] == '%')

                qemu_chr_fe_write(chr, (uint8_t *)ebuf, strlen(ebuf));

            else

                qemu_chr_fe_write(chr, (uint8_t *)&mux_help[i][j], 1);

        }

    }

}
