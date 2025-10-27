static void text_console_update(void *opaque, console_ch_t *chardata)

{

    QemuConsole *s = (QemuConsole *) opaque;

    int i, j, src;



    if (s->text_x[0] <= s->text_x[1]) {

        src = (s->y_base + s->text_y[0]) * s->width;

        chardata += s->text_y[0] * s->width;

        for (i = s->text_y[0]; i <= s->text_y[1]; i ++)

            for (j = 0; j < s->width; j ++, src ++)

                console_write_ch(chardata ++, s->cells[src].ch |

                                (s->cells[src].t_attrib.fgcol << 12) |

                                (s->cells[src].t_attrib.bgcol << 8) |

                                (s->cells[src].t_attrib.bold << 21));

        dpy_text_update(s, s->text_x[0], s->text_y[0],

                        s->text_x[1] - s->text_x[0], i - s->text_y[0]);

        s->text_x[0] = s->width;

        s->text_y[0] = s->height;

        s->text_x[1] = 0;

        s->text_y[1] = 0;

    }

    if (s->cursor_invalidate) {

        dpy_text_cursor(s, s->x, s->y);

        s->cursor_invalidate = 0;

    }

}
