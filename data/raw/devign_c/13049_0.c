static uint32_t ide_ioport_read(void *opaque, uint32_t addr1)

{

    IDEState *ide_if = opaque;

    IDEState *s = ide_if->cur_drive;

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

        if (!ide_if[0].bs && !ide_if[1].bs)

            ret = 0;

        else if (!hob)

            ret = s->error;

	else

	    ret = s->hob_feature;

        break;

    case 2:

        if (!ide_if[0].bs && !ide_if[1].bs)

            ret = 0;

        else if (!hob)

            ret = s->nsector & 0xff;

	else

	    ret = s->hob_nsector;

        break;

    case 3:

        if (!ide_if[0].bs && !ide_if[1].bs)

            ret = 0;

        else if (!hob)

            ret = s->sector;

	else

	    ret = s->hob_sector;

        break;

    case 4:

        if (!ide_if[0].bs && !ide_if[1].bs)

            ret = 0;

        else if (!hob)

            ret = s->lcyl;

	else

	    ret = s->hob_lcyl;

        break;

    case 5:

        if (!ide_if[0].bs && !ide_if[1].bs)

            ret = 0;

        else if (!hob)

            ret = s->hcyl;

	else

	    ret = s->hob_hcyl;

        break;

    case 6:

        if (!ide_if[0].bs && !ide_if[1].bs)

            ret = 0;

        else

            ret = s->select;

        break;

    default:

    case 7:

        if ((!ide_if[0].bs && !ide_if[1].bs) ||

            (s != ide_if && !s->bs))

            ret = 0;

        else

            ret = s->status;

        qemu_irq_lower(s->irq);

        break;

    }

#ifdef DEBUG_IDE

    printf("ide: read addr=0x%x val=%02x\n", addr1, ret);

#endif

    return ret;

}
