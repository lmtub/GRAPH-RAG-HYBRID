static void set_lcd_pixel(musicpal_lcd_state *s, int x, int y, int col)

{

    int dx, dy;



    for (dy = 0; dy < 3; dy++)

        for (dx = 0; dx < 3; dx++) {

            s->ds->data[(x*3 + dx + (y*3 + dy) * 128*3) * 4 + 0] =

                scale_lcd_color(col);

            s->ds->data[(x*3 + dx + (y*3 + dy) * 128*3) * 4 + 1] =

                scale_lcd_color(col >> 8);

            s->ds->data[(x*3 + dx + (y*3 + dy) * 128*3) * 4 + 2] =

                scale_lcd_color(col >> 16);

        }

}
