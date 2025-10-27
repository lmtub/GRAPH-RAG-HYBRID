static void intel_hda_response(HDACodecDevice *dev, bool solicited, uint32_t response)

{

    HDACodecBus *bus = HDA_BUS(dev->qdev.parent_bus);

    IntelHDAState *d = container_of(bus, IntelHDAState, codecs);

    hwaddr addr;

    uint32_t wp, ex;



    if (d->ics & ICH6_IRS_BUSY) {

        dprint(d, 2, "%s: [irr] response 0x%x, cad 0x%x\n",

               __FUNCTION__, response, dev->cad);

        d->irr = response;

        d->ics &= ~(ICH6_IRS_BUSY | 0xf0);

        d->ics |= (ICH6_IRS_VALID | (dev->cad << 4));

        return;

    }



    if (!(d->rirb_ctl & ICH6_RBCTL_DMA_EN)) {

        dprint(d, 1, "%s: rirb dma disabled, drop codec response\n", __FUNCTION__);

        return;

    }



    ex = (solicited ? 0 : (1 << 4)) | dev->cad;

    wp = (d->rirb_wp + 1) & 0xff;

    addr = intel_hda_addr(d->rirb_lbase, d->rirb_ubase);

    stl_le_pci_dma(&d->pci, addr + 8*wp, response);

    stl_le_pci_dma(&d->pci, addr + 8*wp + 4, ex);

    d->rirb_wp = wp;



    dprint(d, 2, "%s: [wp 0x%x] response 0x%x, extra 0x%x\n",

           __FUNCTION__, wp, response, ex);



    d->rirb_count++;

    if (d->rirb_count == d->rirb_cnt) {

        dprint(d, 2, "%s: rirb count reached (%d)\n", __FUNCTION__, d->rirb_count);

        if (d->rirb_ctl & ICH6_RBCTL_IRQ_EN) {

            d->rirb_sts |= ICH6_RBSTS_IRQ;

            intel_hda_update_irq(d);

        }

    } else if ((d->corb_rp & 0xff) == d->corb_wp) {

        dprint(d, 2, "%s: corb ring empty (%d/%d)\n", __FUNCTION__,

               d->rirb_count, d->rirb_cnt);

        if (d->rirb_ctl & ICH6_RBCTL_IRQ_EN) {

            d->rirb_sts |= ICH6_RBSTS_IRQ;

            intel_hda_update_irq(d);

        }

    }

}
