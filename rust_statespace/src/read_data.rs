use std::{
    env,
    error::Error,
    ffi::OsString,
    fs::File,
    process,
    f64
};

use ndarray::{Array1, Array2, array};

// This module should:
// 1. A load data function
// 1.1. Check if file exists in data directory
// 1.2. Check if filetype is csv
// 1.3. Load data into csv

#[allow(non_snake_case)]
pub fn load_csv(file_path: String) -> Result<(Array1<f64>, Array1<f64>), Box<dyn Error>> {
    
    // read file
    let file = File::open(file_path)?;

    // create reader object that reads from filepath
    let mut rdr = csv::Reader::from_reader(file);

    // create object to store data
    let mut x = vec![];
    let mut y = vec![];

    for result in rdr.records() {
        
        // ?: propagate error upwards if parse returns Err
        let record = result?;

        // nile data in first row
        // ?: propagate error upwards if parse returns Err
        x.push(record[0].parse::<f64>()?);
        y.push(record[1].parse::<f64>()?);
        //println!("{:?}", record);
    };

    // return tuple containing output
    Ok((Array1::from(x), Array1::from(y)))
}