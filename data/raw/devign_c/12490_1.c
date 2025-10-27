static uint64_t vfio_rom_read(void *opaque, hwaddr addr, unsigned size)

{

    VFIODevice *vdev = opaque;

    uint64_t val = ((uint64_t)1 << (size * 8)) - 1;



    /* Load the ROM lazily when the guest tries to read it */

    if (unlikely(!vdev->rom)) {








    memcpy(&val, vdev->rom + addr,

           (addr < vdev->rom_size) ? MIN(size, vdev->rom_size - addr) : 0);



    DPRINTF("%s(%04x:%02x:%02x.%x, 0x%"HWADDR_PRIx", 0x%x) = 0x%"PRIx64"\n",

            __func__, vdev->host.domain, vdev->host.bus, vdev->host.slot,

            vdev->host.function, addr, size, val);



    return val;
