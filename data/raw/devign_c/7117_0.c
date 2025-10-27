static void vnc_write_u8(VncState *vs, uint8_t value)

{

    vnc_write(vs, (char *)&value, 1);

}
