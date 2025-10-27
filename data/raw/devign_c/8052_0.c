uint32_t ide_ioport_read(void *opaque, uint32_t addr1)

{

    IDEBus *bus = opaque;

    IDEState *s = idebus_active_if(bus);

    uint32_t addr;

    int ret, hob;



    addr = addr1 & 7;

    /* FIXME: HOB readback uses bit 7, but it's always set right now */

    //hob = s->select & (1 << 7);

    hob = 0;

    switch(addr) {

    case 0:

        ret = 0xff;

        break;

    case 1:

        if ((!bus->ifs[0].bs && !bus->ifs[1].bs) ||

            (s != bus->ifs && !s->bs))

            ret = 0;

        else if (!hob)

            ret = s->error;

	else

	    ret = s->hob_feature;

        break;

    case 2:

        if (!bus->ifs[0].bs && !bus->ifs[1].bs)

            ret = 0;

        else if (!hob)

            ret = s->nsector & 0xff;

	else

	    ret = s->hob_nsector;

        break;

    case 3:

        if (!bus->ifs[0].bs && !bus->ifs[1].bs)

            ret = 0;

        else if (!hob)

            ret = s->sector;

	else

	    ret = s->hob_sector;

        break;

    case 4:

        if (!bus->ifs[0].bs && !bus->ifs[1].bs)

            ret = 0;

        else if (!hob)

            ret = s->lcyl;

	else

	    ret = s->hob_lcyl;

        break;

    case 5:

        if (!bus->ifs[0].bs && !bus->ifs[1].bs)

            ret = 0;

        else if (!hob)

            ret = s->hcyl;

	else

	    ret = s->hob_hcyl;

        break;

    case 6:

        if (!bus->ifs[0].bs && !bus->ifs[1].bs)

            ret = 0;

        else

            ret = s->select;

        break;

    default:

    case 7:

        if ((!bus->ifs[0].bs && !bus->ifs[1].bs) ||

            (s != bus->ifs && !s->bs))

            ret = 0;

        else

            ret = s->status;

        qemu_irq_lower(bus->irq);

        break;

    }

#ifdef DEBUG_IDE

    printf("ide: read addr=0x%x val=%02x\n", addr1, ret);

#endif

    return ret;

}
