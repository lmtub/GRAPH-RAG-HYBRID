static int vga_initfn(ISADevice *dev)

{

    ISACirrusVGAState *d = DO_UPCAST(ISACirrusVGAState, dev, dev);

    VGACommonState *s = &d->cirrus_vga.vga;



    vga_common_init(s);

    cirrus_init_common(&d->cirrus_vga, CIRRUS_ID_CLGD5430, 0,

                       isa_address_space(dev), isa_address_space_io(dev));

    s->con = graphic_console_init(s->update, s->invalidate,

                                  s->screen_dump, s->text_update,

                                  s);

    rom_add_vga(VGABIOS_CIRRUS_FILENAME);

    /* XXX ISA-LFB support */

    /* FIXME not qdev yet */

    return 0;

}
