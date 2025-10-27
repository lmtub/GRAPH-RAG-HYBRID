int i2c_start_transfer(I2CBus *bus, uint8_t address, int recv)

{

    BusChild *kid;

    I2CSlaveClass *sc;

    I2CNode *node;



    if (address == I2C_BROADCAST) {

        /*

         * This is a broadcast, the current_devs will be all the devices of the

         * bus.

         */

        bus->broadcast = true;

    }



    QTAILQ_FOREACH(kid, &bus->qbus.children, sibling) {

        DeviceState *qdev = kid->child;

        I2CSlave *candidate = I2C_SLAVE(qdev);

        if ((candidate->address == address) || (bus->broadcast)) {

            node = g_malloc(sizeof(struct I2CNode));

            node->elt = candidate;

            QLIST_INSERT_HEAD(&bus->current_devs, node, next);

            if (!bus->broadcast) {

                break;

            }

        }

    }



    if (QLIST_EMPTY(&bus->current_devs)) {

        return 1;

    }



    QLIST_FOREACH(node, &bus->current_devs, next) {

        sc = I2C_SLAVE_GET_CLASS(node->elt);

        /* If the bus is already busy, assume this is a repeated

           start condition.  */

        if (sc->event) {

            sc->event(node->elt, recv ? I2C_START_RECV : I2C_START_SEND);

        }

    }

    return 0;

}
