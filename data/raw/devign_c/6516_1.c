static void bus_set_realized(Object *obj, bool value, Error **errp)

{

    BusState *bus = BUS(obj);

    BusClass *bc = BUS_GET_CLASS(bus);

    Error *local_err = NULL;



    if (value && !bus->realized) {

        if (bc->realize) {

            bc->realize(bus, &local_err);

        }

    } else if (!value && bus->realized) {

        if (bc->unrealize) {

            bc->unrealize(bus, &local_err);

        }

    }



    if (local_err != NULL) {

        error_propagate(errp, local_err);

        return;

    }



    bus->realized = value;

}
