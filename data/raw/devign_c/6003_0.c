static void do_sendkey(const char *string)

{

    uint8_t keycodes[16];

    int nb_keycodes = 0;

    char keyname_buf[16];

    char *separator;

    int keyname_len, keycode, i;



    while (1) {

        separator = strchr(string, '-');

        keyname_len = separator ? separator - string : strlen(string);

        if (keyname_len > 0) {

            pstrcpy(keyname_buf, sizeof(keyname_buf), string);

            if (keyname_len > sizeof(keyname_buf) - 1) {

                term_printf("invalid key: '%s...'\n", keyname_buf);

                return;

            }

            if (nb_keycodes == sizeof(keycodes)) {

                term_printf("too many keys\n");

                return;

            }

            keyname_buf[keyname_len] = 0;

            keycode = get_keycode(keyname_buf);

            if (keycode < 0) {

                term_printf("unknown key: '%s'\n", keyname_buf);

                return;

            }

            keycodes[nb_keycodes++] = keycode;

        }

        if (!separator)

            break;

        string = separator + 1;

    }

    /* key down events */

    for(i = 0; i < nb_keycodes; i++) {

        keycode = keycodes[i];

        if (keycode & 0x80)

            kbd_put_keycode(0xe0);

        kbd_put_keycode(keycode & 0x7f);

    }

    /* key up events */

    for(i = nb_keycodes - 1; i >= 0; i--) {

        keycode = keycodes[i];

        if (keycode & 0x80)

            kbd_put_keycode(0xe0);

        kbd_put_keycode(keycode | 0x80);

    }

}
