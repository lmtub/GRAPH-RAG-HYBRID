static inline void vmsvga_fill_rect(struct vmsvga_state_s *s,

                uint32_t c, int x, int y, int w, int h)

{

    DisplaySurface *surface = qemu_console_surface(s->vga.con);

    int bypl = surface_stride(surface);

    int width = surface_bytes_per_pixel(surface) * w;

    int line = h;

    int column;

    uint8_t *fst;

    uint8_t *dst;

    uint8_t *src;

    uint8_t col[4];



    col[0] = c;

    col[1] = c >> 8;

    col[2] = c >> 16;

    col[3] = c >> 24;



    fst = s->vga.vram_ptr + surface_bytes_per_pixel(surface) * x + bypl * y;



    if (line--) {

        dst = fst;

        src = col;

        for (column = width; column > 0; column--) {

            *(dst++) = *(src++);

            if (src - col == surface_bytes_per_pixel(surface)) {

                src = col;

            }

        }

        dst = fst;

        for (; line > 0; line--) {

            dst += bypl;

            memcpy(dst, fst, width);

        }

    }



    vmsvga_update_rect_delayed(s, x, y, w, h);

}
