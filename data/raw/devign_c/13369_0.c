static bool sensor_type_is_dr(uint32_t sensor_type)

{

    switch (sensor_type) {

    case RTAS_SENSOR_TYPE_ISOLATION_STATE:

    case RTAS_SENSOR_TYPE_DR:

    case RTAS_SENSOR_TYPE_ALLOCATION_STATE:

        return true;

    }



    return false;

}
