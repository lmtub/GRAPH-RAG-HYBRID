static int64_t read_ts(char **line, int *duration)

{

    int64_t start, end;



    if (sscanf(*line, "%"SCNd64",%"SCNd64, &start, &end) == 2) {

        *line += strcspn(*line, "\"") + 1;

        *duration = end - start;

        return start;

    }

    return AV_NOPTS_VALUE;

}
