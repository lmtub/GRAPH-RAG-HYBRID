uint32_t ide_status_read(void *opaque, uint32_t addr)

{

    IDEBus *bus = opaque;

    IDEState *s = idebus_active_if(bus);

    int ret;



    if ((!bus->ifs[0].bs && !bus->ifs[1].bs) ||

        (s != bus->ifs && !s->bs))

        ret = 0;

    else

        ret = s->status;

#ifdef DEBUG_IDE

    printf("ide: read status addr=0x%x val=%02x\n", addr, ret);

#endif

    return ret;

}
