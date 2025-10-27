static void vga_screen_dump_blank(VGAState *s, const char *filename)

{

    FILE *f;

    unsigned int y, x, w, h;



    w = s->last_scr_width * sizeof(uint32_t);

    h = s->last_scr_height;



    f = fopen(filename, "wb");

    if (!f)

        return;

    fprintf(f, "P6\n%d %d\n%d\n", w, h, 255);

    for (y = 0; y < h; y++) {

        for (x = 0; x < w; x++) {

            fputc(0, f);

        }

    }

    fclose(f);

}
