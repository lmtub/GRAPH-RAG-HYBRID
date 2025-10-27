int fw_cfg_add_callback(void *opaque, uint16_t key, FWCfgCallback callback,

                        void *callback_opaque, uint8_t *data, size_t len)

{

    FWCfgState *s = opaque;

    int arch = !!(key & FW_CFG_ARCH_LOCAL);



    key &= FW_CFG_ENTRY_MASK;



    if (key >= FW_CFG_MAX_ENTRY || !(key & FW_CFG_WRITE_CHANNEL)

        || len > 65535)

        return 0;



    s->entries[arch][key].data = data;

    s->entries[arch][key].len = len;

    s->entries[arch][key].callback_opaque = callback_opaque;

    s->entries[arch][key].callback = callback;



    return 1;

}
