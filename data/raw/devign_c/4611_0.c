static void ide_atapi_identify(IDEState *s)

{

    uint16_t *p;



    if (s->identify_set) {

	memcpy(s->io_buffer, s->identify_data, sizeof(s->identify_data));

	return;

    }



    memset(s->io_buffer, 0, 512);

    p = (uint16_t *)s->io_buffer;

    /* Removable CDROM, 50us response, 12 byte packets */

    put_le16(p + 0, (2 << 14) | (5 << 8) | (1 << 7) | (2 << 5) | (0 << 0));

    padstr((char *)(p + 10), s->drive_serial_str, 20); /* serial number */

    put_le16(p + 20, 3); /* buffer type */

    put_le16(p + 21, 512); /* cache size in sectors */

    put_le16(p + 22, 4); /* ecc bytes */

    padstr((char *)(p + 23), s->version, 8); /* firmware version */

    padstr((char *)(p + 27), "QEMU DVD-ROM", 40); /* model */

    put_le16(p + 48, 1); /* dword I/O (XXX: should not be set on CDROM) */

#ifdef USE_DMA_CDROM

    put_le16(p + 49, 1 << 9 | 1 << 8); /* DMA and LBA supported */

    put_le16(p + 53, 7); /* words 64-70, 54-58, 88 valid */

    put_le16(p + 62, 7);  /* single word dma0-2 supported */

    put_le16(p + 63, 7);  /* mdma0-2 supported */

#else

    put_le16(p + 49, 1 << 9); /* LBA supported, no DMA */

    put_le16(p + 53, 3); /* words 64-70, 54-58 valid */

    put_le16(p + 63, 0x103); /* DMA modes XXX: may be incorrect */

#endif

    put_le16(p + 64, 3); /* pio3-4 supported */

    put_le16(p + 65, 0xb4); /* minimum DMA multiword tx cycle time */

    put_le16(p + 66, 0xb4); /* recommended DMA multiword tx cycle time */

    put_le16(p + 67, 0x12c); /* minimum PIO cycle time without flow control */

    put_le16(p + 68, 0xb4); /* minimum PIO cycle time with IORDY flow control */



    put_le16(p + 71, 30); /* in ns */

    put_le16(p + 72, 30); /* in ns */



    if (s->ncq_queues) {

        put_le16(p + 75, s->ncq_queues - 1);

        /* NCQ supported */

        put_le16(p + 76, (1 << 8));

    }



    put_le16(p + 80, 0x1e); /* support up to ATA/ATAPI-4 */

#ifdef USE_DMA_CDROM

    put_le16(p + 88, 0x3f | (1 << 13)); /* udma5 set and supported */

#endif

    memcpy(s->identify_data, p, sizeof(s->identify_data));

    s->identify_set = 1;

}
