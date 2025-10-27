static void curses_refresh(DisplayChangeListener *dcl)

{

    int chr, nextchr, keysym, keycode, keycode_alt;



    curses_winch_check();



    if (invalidate) {

        clear();

        refresh();

        curses_calc_pad();

        graphic_hw_invalidate(NULL);

        invalidate = 0;

    }



    graphic_hw_text_update(NULL, screen);



    nextchr = ERR;

    while (1) {

        /* while there are any pending key strokes to process */

        if (nextchr == ERR)

            chr = getch();

        else {

            chr = nextchr;

            nextchr = ERR;

        }



        if (chr == ERR)

            break;



#ifdef KEY_RESIZE

        /* this shouldn't occur when we use a custom SIGWINCH handler */

        if (chr == KEY_RESIZE) {

            clear();

            refresh();

            curses_calc_pad();

            curses_update(dcl, 0, 0, width, height);

            continue;

        }

#endif



        keycode = curses2keycode[chr];

        keycode_alt = 0;



        /* alt key */

        if (keycode == 1) {

            nextchr = getch();



            if (nextchr != ERR) {

                chr = nextchr;

                keycode_alt = ALT;

                keycode = curses2keycode[nextchr];

                nextchr = ERR;



                if (keycode != -1) {

                    keycode |= ALT;



                    /* process keys reserved for qemu */

                    if (keycode >= QEMU_KEY_CONSOLE0 &&

                            keycode < QEMU_KEY_CONSOLE0 + 9) {

                        erase();

                        wnoutrefresh(stdscr);

                        console_select(keycode - QEMU_KEY_CONSOLE0);



                        invalidate = 1;

                        continue;

                    }

                }

            }

        }



        if (kbd_layout) {

            keysym = -1;

            if (chr < CURSES_KEYS)

                keysym = curses2keysym[chr];



            if (keysym == -1) {

                if (chr < ' ') {

                    keysym = chr + '@';

                    if (keysym >= 'A' && keysym <= 'Z')

                        keysym += 'a' - 'A';

                    keysym |= KEYSYM_CNTRL;

                } else

                    keysym = chr;

            }



            keycode = keysym2scancode(kbd_layout, keysym & KEYSYM_MASK);

            if (keycode == 0)

                continue;



            keycode |= (keysym & ~KEYSYM_MASK) >> 16;

            keycode |= keycode_alt;

        }



        if (keycode == -1)

            continue;



        if (qemu_console_is_graphic(NULL)) {

            /* since terminals don't know about key press and release

             * events, we need to emit both for each key received */

            if (keycode & SHIFT) {

                qemu_input_event_send_key_number(NULL, SHIFT_CODE, true);

                qemu_input_event_send_key_delay(0);

            }

            if (keycode & CNTRL) {

                qemu_input_event_send_key_number(NULL, CNTRL_CODE, true);

                qemu_input_event_send_key_delay(0);

            }

            if (keycode & ALT) {

                qemu_input_event_send_key_number(NULL, ALT_CODE, true);

                qemu_input_event_send_key_delay(0);

            }

            if (keycode & ALTGR) {

                qemu_input_event_send_key_number(NULL, GREY | ALT_CODE, true);

                qemu_input_event_send_key_delay(0);

            }



            qemu_input_event_send_key_number(NULL, keycode & KEY_MASK, true);

            qemu_input_event_send_key_delay(0);

            qemu_input_event_send_key_number(NULL, keycode & KEY_MASK, false);

            qemu_input_event_send_key_delay(0);



            if (keycode & ALTGR) {

                qemu_input_event_send_key_number(NULL, GREY | ALT_CODE, false);

                qemu_input_event_send_key_delay(0);

            }

            if (keycode & ALT) {

                qemu_input_event_send_key_number(NULL, ALT_CODE, false);

                qemu_input_event_send_key_delay(0);

            }

            if (keycode & CNTRL) {

                qemu_input_event_send_key_number(NULL, CNTRL_CODE, false);

                qemu_input_event_send_key_delay(0);

            }

            if (keycode & SHIFT) {

                qemu_input_event_send_key_number(NULL, SHIFT_CODE, false);

                qemu_input_event_send_key_delay(0);

            }

        } else {

            keysym = curses2qemu[chr];

            if (keysym == -1)

                keysym = chr;



            kbd_put_keysym(keysym);

        }

    }

}
