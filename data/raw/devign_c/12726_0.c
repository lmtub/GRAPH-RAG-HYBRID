static void do_sendkey(Monitor *mon, const QDict *qdict)

{

    char keyname_buf[16];

    char *separator;

    int keyname_len, keycode, i, idx;

    const char *keys = qdict_get_str(qdict, "keys");

    int has_hold_time = qdict_haskey(qdict, "hold-time");

    int hold_time = qdict_get_try_int(qdict, "hold-time", -1);



    if (nb_pending_keycodes > 0) {

        qemu_del_timer(key_timer);

        release_keys(NULL);

    }

    if (!has_hold_time)

        hold_time = 100;

    i = 0;

    while (1) {

        separator = strchr(keys, '-');

        keyname_len = separator ? separator - keys : strlen(keys);

        if (keyname_len > 0) {

            pstrcpy(keyname_buf, sizeof(keyname_buf), keys);

            if (keyname_len > sizeof(keyname_buf) - 1) {

                monitor_printf(mon, "invalid key: '%s...'\n", keyname_buf);

                return;

            }

            if (i == MAX_KEYCODES) {

                monitor_printf(mon, "too many keys\n");

                return;

            }



            /* Be compatible with old interface, convert user inputted "<" */

            if (!strncmp(keyname_buf, "<", 1) && keyname_len == 1) {

                pstrcpy(keyname_buf, sizeof(keyname_buf), "less");

                keyname_len = 4;

            }



            keyname_buf[keyname_len] = 0;



            idx = index_from_key(keyname_buf);

            if (idx == Q_KEY_CODE_MAX) {

                monitor_printf(mon, "invalid parameter: %s\n", keyname_buf);

                return;

            }



            keycode = key_defs[idx];

            if (keycode < 0) {

                monitor_printf(mon, "unknown key: '%s'\n", keyname_buf);

                return;

            }

            keycodes[i++] = keycode;

        }

        if (!separator)

            break;

        keys = separator + 1;

    }

    nb_pending_keycodes = i;

    /* key down events */

    for (i = 0; i < nb_pending_keycodes; i++) {

        keycode = keycodes[i];

        if (keycode & 0x80)

            kbd_put_keycode(0xe0);

        kbd_put_keycode(keycode & 0x7f);

    }

    /* delayed key up events */

    qemu_mod_timer(key_timer, qemu_get_clock_ns(vm_clock) +

                   muldiv64(get_ticks_per_sec(), hold_time, 1000));

}
