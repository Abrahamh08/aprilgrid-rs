use core::f32;
use std::{
    collections::{HashMap, HashSet},
    f32::consts::PI,
    ops::BitXor,
};

use crate::image_util::GrayImagef32;
use crate::saddle::Saddle;
use crate::{image_util, tag_families};
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use itertools::Itertools;
use kiddo::{KdTree, SquaredEuclidean};

pub struct TagDetector {
    edge: u8,
    border: u8,
    hamming_distance: u8,
    code_list: Vec<u64>,
    detector_params: DetectorParams,
}

pub struct DetectorParams {
    pub tag_spacing_ratio: f32,
    pub min_saddle_angle: f32,
    pub max_saddle_angle: f32,
    pub max_num_of_boards: u8,
}

impl DetectorParams {
    pub fn default_params() -> DetectorParams {
        DetectorParams {
            tag_spacing_ratio: 0.3,
            min_saddle_angle: 30.0,
            max_saddle_angle: 60.0,
            max_num_of_boards: 2,
        }
    }
}
pub fn decode_positions(
    img_w: u32,
    img_h: u32,
    quad_pts: &[(f32, f32)],
    border_bits: u8,
    edge_bits: u8,
    margin: f32,
) -> Option<Vec<(f32, f32)>> {
    if quad_pts.iter().any(|(x, y)| {
        let x = x.round() as u32;
        let y = y.round() as u32;
        x >= img_w || y >= img_h
    }) {
        return None;
    }
    let side_bits = border_bits * 2 + edge_bits;
    let affine_mat = image_util::tag_affine(quad_pts, side_bits, margin);
    Some(
        (border_bits..border_bits + edge_bits)
            .flat_map(|x| {
                (border_bits..border_bits + edge_bits)
                    .map(|y| {
                        let tp = faer::mat![[x as f32], [y as f32], [1.0]];
                        let tt = affine_mat.clone() * tp;
                        (tt[(0, 0)], tt[(1, 0)])
                    })
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
}

pub fn bit_code(
    img: &GrayImage,
    decode_pts: &[(f32, f32)],
    valid_brightness_threshold: u8,
    max_invalid_bit: u32,
) -> Option<u64> {
    let brightness_vec: Vec<u8> = decode_pts
        .iter()
        .filter_map(|(x, y)| {
            let (x, y) = (x.round() as u32, y.round() as u32);
            if x >= img.width() || y >= img.height() {
                None
            } else {
                Some(img.get_pixel(x, y).0[0])
            }
        })
        .collect();
    if brightness_vec.len() != decode_pts.len() {
        return None;
    }
    let (min_b, max_b) = brightness_vec
        .iter()
        .fold((255, 0), |(min_b, max_b), e| (min_b.min(*e), max_b.max(*e)));
    if max_b - min_b < 50 {
        return None;
    }
    let mid_b = ((min_b as f32 + max_b as f32) / 2.0).round() as u8;
    let (bits, invalid_count): (u64, u32) = brightness_vec.iter().rev().enumerate().fold(
        (0u64, 0u32),
        |(acc, invalid_count), (i, b)| {
            let invalid_count =
                if (mid_b as i32 - *b as i32).abs() < valid_brightness_threshold as i32 {
                    invalid_count + 1
                } else {
                    invalid_count
                };
            if *b > mid_b {
                (acc | (1 << i), invalid_count)
            } else {
                (acc, invalid_count)
            }
        },
    );
    if invalid_count > max_invalid_bit {
        None
    } else {
        Some(bits)
    }
}

const fn rotate_bits(bits: u64, edge_bits: u8) -> u64 {
    let edge_bits = edge_bits as usize;
    let mut b = 0u64;
    let mut count = 0;
    let mut r = (edge_bits - 1) as i32;
    while r >= 0 {
        let mut c = 0;
        while c < edge_bits {
            let idx = r as usize + c * edge_bits;
            b |= ((bits >> idx) & 1) << count;
            count += 1;
            c += 1;
        }
        r -= 1;
    }
    b
}

pub fn best_tag(bits: u64, thres: u8, tag_family: &[u64], edge_bits: u8) -> Option<(usize, usize)> {
    let mut bits = bits;
    for rotated in 0..4 {
        let scores: Vec<u32> = tag_family
            .iter()
            .map(|t| t.bitxor(bits).count_ones())
            .collect();
        let (best_idx, best_score) = scores
            .iter()
            .enumerate()
            .reduce(|(best_idx, best_score), (cur_idx, cur_socre)| {
                if cur_socre < best_score {
                    (cur_idx, cur_socre)
                } else {
                    (best_idx, best_score)
                }
            })
            .unwrap();
        if *best_score < thres as u32 {
            // println!("best {} {} rotate {}", best_idx, best_score, rotated);
            return Some((best_idx, rotated));
        } else if rotated == 3 {
            break;
        }
        bits = rotate_bits(bits, edge_bits);
    }
    None
}

fn init_saddle_clusters(h_mat: &GrayImagef32, threshold: f32) -> Vec<Vec<(u32, u32)>> {
    let mut tmp_h_mat = h_mat.clone();
    let mut clusters = Vec::new();
    for r in 1..h_mat.height() - 1 {
        for c in 1..h_mat.width() - 1 {
            let mut cluster = Vec::new();
            image_util::pixel_bfs(&mut tmp_h_mat, &mut cluster, c, r, threshold);
            if !cluster.is_empty() {
                clusters.push(cluster);
            }
        }
    }
    clusters
}

pub struct Tag {
    pub id: u32,
    pub p: [(f32, f32); 4],
}

/// Cubic interpolation using Catmull–Rom.
/// Given four values (v0, v1, v2, v3) and a parameter t in [0, 1],
/// returns the interpolated value.
fn cubic_interp(v0: f32, v1: f32, v2: f32, v3: f32, t: f32) -> f32 {
    let a = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3;
    let b = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3;
    let c = -0.5 * v0 + 0.5 * v2;
    let d = v1;
    a * t * t * t + b * t * t + c * t + d
}

/// Bicubic interpolation sample function.
/// Uses a 4×4 neighborhood around (x, y) (in f32 coordinates)
/// and falls back to bilinear interpolation if too close to the border.
fn bicubic_sample<T>(
    image: &ImageBuffer<Luma<T>, Vec<T>>,
    x: f32,
    y: f32,
) -> f32
where
    T: image::Primitive + Into<f32> + Copy,
{
    let width = image.width() as i32;
    let height = image.height() as i32;

    let x_int = x.floor() as i32;
    let y_int = y.floor() as i32;
    let t = x - x.floor();
    let u = y - y.floor();

    // Fall back to bilinear interpolation if too near the border.
    if x_int < 1 || y_int < 1 || x_int >= width - 2 || y_int >= height - 2 {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        if x0 < 0 || y0 < 0 || x1 >= width || y1 >= height {
            return 0.0;
        }
        let dx = x - x0 as f32;
        let dy = y - y0 as f32;
        let p00: f32 = image.get_pixel(x0 as u32, y0 as u32).0[0].into();
        let p10: f32 = image.get_pixel(x1 as u32, y0 as u32).0[0].into();
        let p01: f32 = image.get_pixel(x0 as u32, y1 as u32).0[0].into();
        let p11: f32 = image.get_pixel(x1 as u32, y1 as u32).0[0].into();
        return (1.0 - dx) * (1.0 - dy) * p00 +
               dx * (1.0 - dy) * p10 +
               (1.0 - dx) * dy * p01 +
               dx * dy * p11;
    }

    let mut arr = [0.0_f32; 4];
    for m in -1..=2 {
        let mut col = [0.0_f32; 4];
        for n in -1..=2 {
            let sample_x = (x_int + n) as u32;
            let sample_y = (y_int + m) as u32;
            col[(n + 1) as usize] = image.get_pixel(sample_x, sample_y).0[0].into();
        }
        arr[(m + 1) as usize] = cubic_interp(col[0], col[1], col[2], col[3], t);
    }
    // Interpolate the four intermediate values along y.
    cubic_interp(arr[0], arr[1], arr[2], arr[3], u)
}

/// Computes the gradient (gx, gy) and Hessian (fxx, fyy, fxy) at (x, y)
/// using central differences with a small step size.
fn compute_gradient_hessian<T>(
    image: &ImageBuffer<Luma<T>, Vec<T>>,
    x: f32,
    y: f32,
) -> (f32, f32, f32, f32, f32)
where
    T: image::Primitive + Into<f32> + Copy,
{
    let h = 1e-1_f32;
    let f_center = bicubic_sample(image, x, y);
    let f_x_plus = bicubic_sample(image, x + h, y);
    let f_x_minus = bicubic_sample(image, x - h, y);
    let f_y_plus = bicubic_sample(image, x, y + h);
    let f_y_minus = bicubic_sample(image, x, y - h);
    let f_xy_pp = bicubic_sample(image, x + h, y + h);
    let f_xy_pm = bicubic_sample(image, x + h, y - h);
    let f_xy_mp = bicubic_sample(image, x - h, y + h);
    let f_xy_mm = bicubic_sample(image, x - h, y - h);

    let gx = (f_x_plus - f_x_minus) / (2.0 * h);
    let gy = (f_y_plus - f_y_minus) / (2.0 * h);
    let fxx = (f_x_plus - 2.0 * f_center + f_x_minus) / (h * h);
    let fyy = (f_y_plus - 2.0 * f_center + f_y_minus) / (h * h);
    let fxy = (f_xy_pp - f_xy_pm - f_xy_mp + f_xy_mm) / (4.0 * h * h);
    (gx, gy, fxx, fyy, fxy)
}

/// Refines initial corner points using Newton–Raphson iteration.
/// The initial corners (in f32) are assumed to be at the top–left of a pixel;
/// they are shifted to the pixel center (by adding 0.5) before refinement.
/// Returns a vector of refined Saddle points.
pub fn rochade_refine<T>(
    image_input: &ImageBuffer<Luma<T>, Vec<T>>,
    initial_corners: &Vec<(f32, f32)>,
) -> Vec<Saddle>
where
    T: image::Primitive + Into<f32> + Copy,
{
    const PIXEL_MOVE_THRESHOLD: f32 = 0.0001;
    const MAX_ITER: usize = 10000;
    let mut refined_corners = Vec::new();

    let width = image_input.width() as f32;
    let height = image_input.height() as f32;

    for &(init_x, init_y) in initial_corners {
        // Start at the center of the pixel.
        let mut x = init_x + 0.5;
        let mut y = init_y + 0.5;
        let mut iter = 0;

        loop {
            iter += 1;
            if iter > MAX_ITER {
                break;
            }
            // Ensure we are within safe bounds for bicubic sampling.
            if x < 1.0 || y < 1.0 || x > width - 3.0 || y > height - 3.0 {
                break;
            }

            let (gx, gy, fxx, fyy, fxy) = compute_gradient_hessian(image_input, x, y);
            let det = fxx * fyy - fxy * fxy;
            if det.abs() < 1e-12 {
                break;
            }
            // Newton–Raphson update: Δ = -H⁻¹ ∇f.
            let delta_x = -(fyy * gx - fxy * gy) / det;
            let delta_y = -(-fxy * gx + fxx * gy) / det;
            if delta_x.abs() < PIXEL_MOVE_THRESHOLD && delta_y.abs() < PIXEL_MOVE_THRESHOLD {
                x += delta_x;
                y += delta_y;
                break;
            }
            x += delta_x;
            y += delta_y;
        }

        // Check if final position is within bounds.
        if x < 1.0 || y < 1.0 || x > width - 3.0 || y > height - 3.0 {
            continue;
        }

        // Recompute Hessian at the refined position.
        let (_gx, _gy, fxx, fyy, fxy) = compute_gradient_hessian(image_input, x, y);
        if fxx * fyy - fxy * fxy >= 0.0 {
            continue;
        }

        // Compute quadratic parameters for saddle characterization.
        let a1 = fxx / 2.0;
        let a2 = fxy;
        let a3 = fyy / 2.0;
        let c5 = (a1 + a3) / 2.0;
        let c4 = (a1 - a3) / 2.0;
        let c3 = a2 / 2.0;
        let k = (c4 * c4 + c3 * c3).sqrt();
        if k.abs() < 1e-12 || c5.abs() >= k {
            continue;
        }
        let phi = (-c5 / k).acos() / 2.0 / PI * 180.0;
        let theta = c3.atan2(c4) / 2.0 / PI * 180.0;

        refined_corners.push(Saddle {
            p: (x, y),
            k,
            theta,
            phi,
        });
    }
    refined_corners
}

impl TagDetector {
    pub fn new(
        tag_family: &tag_families::TagFamily,
        optional_detector_params: Option<DetectorParams>,
    ) -> TagDetector {
        let detector_params = optional_detector_params.unwrap_or(DetectorParams::default_params());
        match tag_family {
            tag_families::TagFamily::T16H5 => TagDetector {
                edge: 4,
                border: 2,
                hamming_distance: 1,
                code_list: tag_families::T16H5.to_vec(),
                detector_params,
            },
            tag_families::TagFamily::T25H7 => TagDetector {
                edge: 5,
                border: 2,
                hamming_distance: 2,
                code_list: tag_families::T25H7.to_vec(),
                detector_params,
            },
            tag_families::TagFamily::T25H9 => TagDetector {
                edge: 5,
                border: 2,
                hamming_distance: 2,
                code_list: tag_families::T25H9.to_vec(),
                detector_params,
            },
            tag_families::TagFamily::T36H11 => TagDetector {
                edge: 6,
                border: 2,
                hamming_distance: 3,
                code_list: tag_families::T36H11.to_vec(),
                detector_params,
            },
            tag_families::TagFamily::T36H11B1 => TagDetector {
                edge: 6,
                border: 1,
                hamming_distance: 3,
                code_list: tag_families::T36H11.to_vec(),
                detector_params,
            },
        }
    }

    pub fn refined_saddle_points(&self, img: &DynamicImage) -> Vec<Saddle> {
        let blur: GrayImagef32 = imageproc::filter::gaussian_blur_f32(&img.to_luma32f(), 1.5);
        let hessian_response_mat = image_util::hessian_response(&blur);
        let min_response = hessian_response_mat
            .to_vec()
            .iter()
            .fold(f32::MAX, |acc, e| acc.min(*e));
        let min_response_threshold = min_response * 0.05;
        let saddle_clusters = init_saddle_clusters(&hessian_response_mat, min_response_threshold);
        let saddle_cluster_centers: Vec<(f32, f32)> = saddle_clusters
            .iter()
            .map(|c| {
                let (sx, sy) = c.iter().fold((0.0, 0.0), |(ax, ay), (ex, ey)| {
                    (ax + *ex as f32, ay + *ey as f32)
                });
                (sx / c.len() as f32, sy / c.len() as f32)
            })
            .collect();
        let saddle_points = rochade_refine(&blur, &saddle_cluster_centers);
        let smax = saddle_points.iter().fold(f32::MIN, |acc, s| acc.max(s.k)) / 10.0;
        let refined: Vec<Saddle> = saddle_points
            .iter()
            .filter_map(|s| {
                if s.k < smax
                    || s.phi < self.detector_params.min_saddle_angle
                    || s.phi > self.detector_params.max_saddle_angle
                {
                    None
                } else {
                    Some(s.to_owned())
                }
            })
            .collect();
        refined
    }

    fn try_decode_quad(
        &self,
        img_grey: &GrayImage,
        quad_points: &[(f32, f32)],
    ) -> Option<(usize, [(f32, f32); 4])> {
        let homo_points_option = decode_positions(
            img_grey.width(),
            img_grey.height(),
            quad_points,
            self.border,
            self.edge,
            0.5,
        );
        if let Some(homo_points) = homo_points_option {
            let bits_option = bit_code(img_grey, &homo_points, 10, 3);
            if let Some(bits) = bits_option {
                let tag_id_option =
                    best_tag(bits, self.hamming_distance, &self.code_list, self.edge);
                if let Some((tag_id, rotation)) = tag_id_option {
                    let mut new_q_pts = quad_points.to_owned();
                    new_q_pts.rotate_left(rotation);
                    new_q_pts.reverse();
                    let refined_arr: [(f32, f32); 4] = new_q_pts.try_into().unwrap();
                    return Some((tag_id, refined_arr));
                }
            }
        }
        None
    }

    #[cfg(feature = "kornia")]
    pub fn detect_kornia<const N: usize>(
        &self,
        img: &kornia::image::Image<u8, N>,
    ) -> HashMap<u32, [(f32, f32); 4]> {
        let dyn_img = match img.num_channels() {
            1 => DynamicImage::ImageLuma8(
                GrayImage::from_vec(
                    img.width() as u32,
                    img.height() as u32,
                    img.clone().0.into_vec(),
                )
                .unwrap(),
            ),
            3 => DynamicImage::ImageRgb8(
                image::RgbImage::from_vec(
                    img.width() as u32,
                    img.height() as u32,
                    img.clone().0.into_vec(),
                )
                .unwrap(),
            ),
            _ => panic!("Only support u8c1 and u8c3"),
        };
        self.detect(&dyn_img)
    }

    pub fn detect(&self, img: &DynamicImage) -> HashMap<u32, [(f32, f32); 4]> {
        let mut detected_tags = HashMap::new();
        let img_grey = img.to_luma8();
        let mut refined = self.refined_saddle_points(img);

        for _ in 0..self.detector_params.max_num_of_boards {
            let best_board_indexes_option = try_find_best_board(&refined);
            if let Some(best_board_indexes) = best_board_indexes_option {
                let mut indexs_to_remove = HashSet::new();
                for quad_indexes in best_board_indexes {
                    let quad_points: Vec<(f32, f32)> =
                        quad_indexes.iter().map(|i| refined[*i].p).collect();
                    if let Some((tag_id, refined_arr)) =
                        self.try_decode_quad(&img_grey, &quad_points)
                    {
                        detected_tags.insert(tag_id as u32, refined_arr);
                        for qi in quad_indexes {
                            indexs_to_remove.insert(qi);
                        }
                    }
                }
                refined = refined
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| {
                        if indexs_to_remove.contains(&i) {
                            None
                        } else {
                            Some(*s)
                        }
                    })
                    .collect();
            }
        }
        detected_tags
    }
}

pub fn init_quads(refined: &[Saddle], s0_idx: usize, tree: &KdTree<f32, 2>) -> Vec<[usize; 4]> {
    let mut out = Vec::new();
    let s0 = refined[s0_idx];
    let nearest = tree.nearest_n::<SquaredEuclidean>(&s0.arr(), 50);
    let mut same_p_idxs = Vec::new();
    let mut diff_p_idxs = Vec::new();
    for n in &nearest[1..] {
        let s = refined[n.item as usize];
        let theta_diff = crate::math_util::theta_distance_degree(s0.theta, s.theta);
        if theta_diff < 5.0 {
            same_p_idxs.push(n.item as usize);
        } else if theta_diff > 80.0 {
            diff_p_idxs.push(n.item as usize);
        }
    }
    for s1_idx in same_p_idxs {
        let s1 = refined[s1_idx];
        for dp in diff_p_idxs.iter().combinations(2) {
            let d0 = refined[*dp[0]];
            let d1 = refined[*dp[1]];
            if !crate::saddle::is_valid_quad(&s0, &d0, &s1, &d1) {
                // if s1_idx == 30 && *dp[1] == 28 && *dp[0] == 60{
                //     panic!("aaaa");
                // }
                continue;
            }
            let v01 = (d0.p.0 - s0.p.0, d0.p.1 - s0.p.1);
            let v02 = (s1.p.0 - s0.p.0, s1.p.1 - s0.p.1);
            let c0 = crate::math_util::cross(&v01, &v02);
            let quad_idxs = if c0 > 0.0 {
                [s0_idx, *dp[0], s1_idx, *dp[1]]
            } else {
                [s0_idx, *dp[1], s1_idx, *dp[0]]
            };
            out.push(quad_idxs);
        }
    }
    out
}

pub fn try_find_best_board(refined: &[Saddle]) -> Option<Vec<[usize; 4]>> {
    if refined.is_empty() {
        return None;
    }
    let entries: Vec<[f32; 2]> = refined.iter().map(|r| r.p.into()).collect();
    // use the kiddo::KdTree type to get up and running quickly with default settings
    let tree: KdTree<f32, 2> = (&entries).into();

    // quad search
    let active_idxs: HashSet<usize> = (0..refined.len()).collect();
    let (mut best_score, mut best_board_option) = (0, None);
    let mut count = 0;
    let mut hm = HashMap::<i32, Vec<usize>>::new();
    refined.iter().enumerate().for_each(|(i, s)| {
        let angle = s.theta.round() as i32;
        if let std::collections::hash_map::Entry::Vacant(e) = hm.entry(angle) {
            e.insert(vec![i]);
        } else {
            hm.get_mut(&angle).unwrap().push(i);
        }
    });
    let mut s0_idxs: Vec<usize> = hm
        .iter()
        .sorted_by(|a, b| a.1.len().cmp(&b.1.len()))
        .next_back()
        .unwrap()
        .1
        .to_owned();
    while !s0_idxs.is_empty() && count < 30 {
        let s0_idx = s0_idxs.pop().unwrap();
        let quads = init_quads(refined, s0_idx, &tree);
        for q in quads {
            let board = crate::board::Board::new(refined, &active_idxs, &q, 0.3, &tree);
            if board.score > best_score {
                best_score = board.score;
                best_board_option = Some(board);
            }
        }
        if best_score >= 36 {
            break;
        }
        count += 1;
    }
    if let Some(mut best_board) = best_board_option {
        best_board.try_fix_missing();
        let tag_idxs: Vec<[usize; 4]> = best_board.all_tag_indexes();
        Some(tag_idxs)
    } else {
        None
    }
}
