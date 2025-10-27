DisplaySurface *qemu_create_displaysurface_from(int width, int height, int bpp,

                                                int linesize, uint8_t *data)

{

    DisplaySurface *surface = g_new0(DisplaySurface, 1);



    surface->pf = qemu_default_pixelformat(bpp);



    surface->format = qemu_pixman_get_format(&surface->pf);

    assert(surface->format != 0);

    surface->image = pixman_image_create_bits(surface->format,

                                              width, height,

                                              (void *)data, linesize);

    assert(surface->image != NULL);



#ifdef HOST_WORDS_BIGENDIAN

    surface->flags = QEMU_BIG_ENDIAN_FLAG;

#endif



    return surface;

}
