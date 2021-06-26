const simplify = require("simplify-geometry");
const fs = require("fs");

const json_file = "modified_metadata.json";
const json_data = JSON.parse(fs.readFileSync(json_file).toString());

json_data.plots.forEach((plot, idx) => {
    plot.segmentation.forEach((segmentation, idy) => {
        simplified_segmentation = simplify(segmentation,3)
        json_data.plots[idx].segmentation[idy] = simplified_segmentation
        // console.log(segmentation.length)
        // console.log(json_data.plots[idx].segmentation[idy].length, '\n')
    });
});

fs.writeFileSync('optimized_metadata.json', JSON.stringify(json_data));
