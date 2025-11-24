use actix_web::{ post, web, App, HttpResponse, HttpServer, Responder };
use serde::{ Deserialize, Serialize };
use tract_onnx::prelude::*;

#[derive(Deserialize)]
struct PredictRequest {
    input: Vec<f32>,
}

#[derive(Serialize)]
struct PredictResponse {
    prediction: Vec<f32>,
}

#[post("/predict")]
async fn predict(req: web::Json<PredictRequest>) -> impl Responder {
    log::info!("Received /predict request");

    if req.input.is_empty() {
        log::warn!("Rejecting empty input array");
        return HttpResponse::BadRequest().body("Input array cannot be empty");
    }

    // Try loading model
    let model = match tract_onnx::onnx().model_for_path("model.onnx") {
        Ok(m) => m,
        Err(e) => {
            log::error!("Model load error: {}", e);
            return HttpResponse::InternalServerError().body(format!("Model load error: {}", e));
        }
    };

    // Build input shape
    let input_shape = tvec![req.input.len()];

    let model = match
        model.with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))
    {
        Ok(m) => m,
        Err(e) => {
            log::error!("Failed to set input shape: {}", e);
            return HttpResponse::InternalServerError().body(format!("Model shape error: {}", e));
        }
    };

    let model = match model.into_optimized().and_then(|m| m.into_runnable()) {
        Ok(m) => m,
        Err(e) => {
            log::error!("Model optimization error: {}", e);
            return HttpResponse::InternalServerError().body(
                format!("Model optimization error: {}", e)
            );
        }
    };

    // Prepare input tensor
    let input_tensor: Tensor = tract_ndarray::Array1::from(req.input.clone()).into_tensor();

    // Run inference
    let outputs = match model.run(tvec![input_tensor.into_tvalue()]) {
        Ok(out) => out,
        Err(e) => {
            log::error!("Prediction failed: {:?}", e);
            return HttpResponse::InternalServerError().body(format!("Prediction error: {:?}", e));
        }
    };

    // Extract result
    let output_tensor = outputs[0].to_array_view::<f32>().unwrap();
    let output_vec = output_tensor.iter().cloned().collect::<Vec<f32>>();
    log::info!("Prediction result: {:?}", output_vec);

    HttpResponse::Ok().json(PredictResponse {
        prediction: output_vec,
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    log::info!("Starting Rust ML prediction API on http://localhost:8082");

    HttpServer::new(|| App::new().service(predict))
        .bind(("127.0.0.1", 8082))?
        .run().await
}
