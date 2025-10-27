static void fw_cfg_write(FWCfgState *s, uint8_t value)

{

    int arch = !!(s->cur_entry & FW_CFG_ARCH_LOCAL);

    FWCfgEntry *e = &s->entries[arch][s->cur_entry & FW_CFG_ENTRY_MASK];



    FW_CFG_DPRINTF("write %d\n", value);



    if (s->cur_entry & FW_CFG_WRITE_CHANNEL && s->cur_offset < e->len) {

        e->data[s->cur_offset++] = value;

        if (s->cur_offset == e->len) {

            e->callback(e->callback_opaque, e->data);

            s->cur_offset = 0;

        }

    }

}
