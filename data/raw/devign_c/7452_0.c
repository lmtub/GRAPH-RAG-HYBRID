int usb_desc_msos(const USBDesc *desc,  USBPacket *p,

                  int index, uint8_t *dest, size_t len)

{

    void *buf = g_malloc0(4096);

    int length = 0;



    switch (index) {

    case 0x0004:

        length = usb_desc_msos_compat(desc, buf);

        break;

    case 0x0005:

        length = usb_desc_msos_prop(desc, buf);

        break;

    }



    if (length > len) {

        length = len;

    }

    memcpy(dest, buf, length);

    free(buf);



    p->actual_length = length;

    return 0;

}
