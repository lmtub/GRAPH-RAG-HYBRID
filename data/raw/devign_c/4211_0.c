static void vfio_bar_write(void *opaque, hwaddr addr,

                           uint64_t data, unsigned size)

{

    VFIOBAR *bar = opaque;

    union {

        uint8_t byte;

        uint16_t word;

        uint32_t dword;

        uint64_t qword;

    } buf;



    switch (size) {

    case 1:

        buf.byte = data;

        break;

    case 2:

        buf.word = cpu_to_le16(data);

        break;

    case 4:

        buf.dword = cpu_to_le32(data);

        break;

    default:

        hw_error("vfio: unsupported write size, %d bytes\n", size);

        break;

    }



    if (pwrite(bar->fd, &buf, size, bar->fd_offset + addr) != size) {

        error_report("%s(,0x%"HWADDR_PRIx", 0x%"PRIx64", %d) failed: %m",

                     __func__, addr, data, size);

    }



    DPRINTF("%s(BAR%d+0x%"HWADDR_PRIx", 0x%"PRIx64", %d)\n",

            __func__, bar->nr, addr, data, size);



    /*

     * A read or write to a BAR always signals an INTx EOI.  This will

     * do nothing if not pending (including not in INTx mode).  We assume

     * that a BAR access is in response to an interrupt and that BAR

     * accesses will service the interrupt.  Unfortunately, we don't know

     * which access will service the interrupt, so we're potentially

     * getting quite a few host interrupts per guest interrupt.

     */

    vfio_eoi(container_of(bar, VFIODevice, bars[bar->nr]));

}
