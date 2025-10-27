static void vga_screen_dump(void *opaque, const char *filename)

{

    VGAState *s = (VGAState *)opaque;



    if (!(s->ar_index & 0x20))

        vga_screen_dump_blank(s, filename);

    else if (s->gr[6] & 1)

        vga_screen_dump_graphic(s, filename);

    else

        vga_screen_dump_text(s, filename);

}
