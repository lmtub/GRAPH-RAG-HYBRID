static uint64_t vmxnet3_get_command_status(VMXNET3State *s)

{

    uint64_t ret;



    switch (s->last_command) {

    case VMXNET3_CMD_ACTIVATE_DEV:

        ret = (s->device_active) ? 0 : -1;

        VMW_CFPRN("Device active: %" PRIx64, ret);

        break;



    case VMXNET3_CMD_RESET_DEV:

    case VMXNET3_CMD_QUIESCE_DEV:

    case VMXNET3_CMD_GET_QUEUE_STATUS:

        ret = 0;

        break;



    case VMXNET3_CMD_GET_LINK:

        ret = s->link_status_and_speed;

        VMW_CFPRN("Link and speed: %" PRIx64, ret);

        break;



    case VMXNET3_CMD_GET_PERM_MAC_LO:

        ret = vmxnet3_get_mac_low(&s->perm_mac);

        break;



    case VMXNET3_CMD_GET_PERM_MAC_HI:

        ret = vmxnet3_get_mac_high(&s->perm_mac);

        break;



    case VMXNET3_CMD_GET_CONF_INTR:

        ret = vmxnet3_get_interrupt_config(s);

        break;



    case VMXNET3_CMD_GET_ADAPTIVE_RING_INFO:

        ret = VMXNET3_DISABLE_ADAPTIVE_RING;

        break;



    default:

        VMW_WRPRN("Received request for unknown command: %x", s->last_command);

        ret = -1;

        break;

    }



    return ret;

}
