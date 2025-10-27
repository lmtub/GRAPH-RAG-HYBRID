void omap_mpuio_key(struct omap_mpuio_s *s, int row, int col, int down)

{

    if (row >= 5 || row < 0)

        hw_error("%s: No key %i-%i\n", __FUNCTION__, col, row);



    if (down)

        s->buttons[row] |= 1 << col;

    else

        s->buttons[row] &= ~(1 << col);



    omap_mpuio_kbd_update(s);

}
