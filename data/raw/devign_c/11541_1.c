static inline void vmsvga_check_size(struct vmsvga_state_s *s)

{

    DisplaySurface *surface = qemu_console_surface(s->vga.con);



    if (s->new_width != surface_width(surface) ||

        s->new_height != surface_height(surface)) {

        qemu_console_resize(s->vga.con, s->new_width, s->new_height);

        s->invalidated = 1;

    }

}
