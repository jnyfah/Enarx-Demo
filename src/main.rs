#![allow(non_snake_case)]
use rml::knn;
use std::env;
use rml::math;
use std::error::Error;
use std::time::Instant;


type CSVOutput = (Vec<Vec<f64>>, Vec<i32>);

fn parse_csv(data: &str) -> Result<CSVOutput, Box<dyn Error>> {
    let mut out_data: CSVOutput = (Vec::new(), Vec::new());
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(data)?;

    for line in reader.records() {
        let result = line?;
        let mut line_data: (Vec<f64>, i32) = (Vec::new(), 0);
        line_data.1 = (result.get(result.len() - 1).unwrap()).parse()?;
        for i in 0..result.len() - 1 {
            line_data.0.push((result.get(i).unwrap()).parse()?);
        }

        out_data.0.push(line_data.0);
        out_data.1.push(line_data.1);
    }
    Ok(out_data)
}


fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    println!("{}", args[2]);
    println!("{}", args[1]);

    // Format: (Vectors of each feature, Vector of class label)
    let training_data = parse_csv(&args[1])?;
    let testing_data = parse_csv(&args[2])?;

    let start = Instant::now();

    let knn = knn::KNN::new(
        5,
        training_data.0,
        training_data.1,
        None,
        Some(math::norm::Norm::L2),
    );

    let pred: Vec<i32> = testing_data.0.iter().map(|x| knn.predict(x)).collect();

    let num_correct = pred
        .iter()
        .cloned()
        .zip(&testing_data.1)
        .filter(|(a, b)| *a == **b)
        .count();

    println!(
        "Accuracy: {} Runtime: {}s",
        (num_correct as f64) / (pred.len() as f64),
        start.elapsed().as_secs_f64()
    );

    Ok(())
}