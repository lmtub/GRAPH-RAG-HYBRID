static inline void softusb_read_dmem(MilkymistSoftUsbState *s,

        uint32_t offset, uint8_t *buf, uint32_t len)

{

    if (offset + len >= s->dmem_size) {

        error_report("milkymist_softusb: read dmem out of bounds "

                "at offset 0x%x, len %d", offset, len);


        return;

    }



    memcpy(buf, s->dmem_ptr + offset, len);

}