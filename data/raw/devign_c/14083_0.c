void i2c_end_transfer(I2CBus *bus)

{

    I2CSlaveClass *sc;

    I2CNode *node, *next;



    if (QLIST_EMPTY(&bus->current_devs)) {

        return;

    }



    QLIST_FOREACH_SAFE(node, &bus->current_devs, next, next) {

        sc = I2C_SLAVE_GET_CLASS(node->elt);

        if (sc->event) {

            sc->event(node->elt, I2C_FINISH);

        }

        QLIST_REMOVE(node, next);

        g_free(node);

    }

    bus->broadcast = false;

}
