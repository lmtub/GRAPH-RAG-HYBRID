static inline void vmsvga_copy_rect(struct vmsvga_state_s *s,

                int x0, int y0, int x1, int y1, int w, int h)

{

    DisplaySurface *surface = qemu_console_surface(s->vga.con);

    uint8_t *vram = s->vga.vram_ptr;

    int bypl = surface_stride(surface);

    int bypp = surface_bytes_per_pixel(surface);

    int width = bypp * w;

    int line = h;

    uint8_t *ptr[2];



    if (y1 > y0) {

        ptr[0] = vram + bypp * x0 + bypl * (y0 + h - 1);

        ptr[1] = vram + bypp * x1 + bypl * (y1 + h - 1);

        for (; line > 0; line --, ptr[0] -= bypl, ptr[1] -= bypl) {

            memmove(ptr[1], ptr[0], width);

        }

    } else {

        ptr[0] = vram + bypp * x0 + bypl * y0;

        ptr[1] = vram + bypp * x1 + bypl * y1;

        for (; line > 0; line --, ptr[0] += bypl, ptr[1] += bypl) {

            memmove(ptr[1], ptr[0], width);

        }

    }



    vmsvga_update_rect_delayed(s, x1, y1, w, h);

}
