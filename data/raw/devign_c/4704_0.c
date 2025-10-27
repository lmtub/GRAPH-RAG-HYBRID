static void vnc_write_u32(VncState *vs, uint32_t value)

{

    uint8_t buf[4];



    buf[0] = (value >> 24) & 0xFF;

    buf[1] = (value >> 16) & 0xFF;

    buf[2] = (value >>  8) & 0xFF;

    buf[3] = value & 0xFF;



    vnc_write(vs, buf, 4);

}
