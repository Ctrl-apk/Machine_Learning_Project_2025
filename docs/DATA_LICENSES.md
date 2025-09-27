# Data Licenses and Usage

## Mendeley Plant Leaf Diseases Dataset

**Dataset**: Plant Leaf Diseases Dataset  
**DOI**: 10.17632/tywbtsjrjv.1  
**License**: CC0 1.0 Universal (Public Domain)  
**Source**: [Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1)

### License Details
- **CC0 1.0**: This dataset is in the public domain
- **Commercial Use**: ✅ Allowed
- **Modification**: ✅ Allowed
- **Distribution**: ✅ Allowed
- **Attribution**: ❌ Not required (but appreciated)

### Dataset Contents
- **Total Images**: ~61,486
- **Classes**: 39 plant disease classes
- **Crops**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- **Additional**: Background_without_leaves class

### Usage in Sapling ML
This dataset serves as the primary training data for our plant disease classification model. We use:
- Original images for training/validation/test splits
- Augmented images are handled separately to prevent data leakage
- Proper deduplication ensures no image appears in multiple splits

## PlantDoc Dataset

**Dataset**: PlantDoc Dataset  
**License**: MIT License  
**Source**: [GitHub Repository](https://github.com/pratikkayal/PlantDoc-Dataset)

### License Details
- **MIT License**: Permissive open source license
- **Commercial Use**: ✅ Allowed
- **Modification**: ✅ Allowed
- **Distribution**: ✅ Allowed
- **Attribution**: ✅ Required

### Dataset Contents
- **Total Images**: ~2,500
- **Purpose**: Cross-dataset evaluation (out-of-domain testing)
- **Characteristics**: Real field images (not lab conditions)

### Usage in Sapling ML
- Used exclusively for cross-dataset evaluation
- Tests model generalization to real-world field conditions
- Helps assess domain robustness

## Data Processing and Privacy

### Image Processing
- All images are processed locally
- No data is sent to external services during training
- Images are resized and normalized for model input
- Perceptual hashing is used for deduplication

### Privacy Considerations
- No personal information is collected
- Images are of plant leaves only
- No geolocation or metadata is stored
- All processing is done in secure environments

## Citation Requirements

If you use this project or the datasets, please cite:

### Mendeley Dataset
```
@dataset{plant_diseases_2019,
  title={Plant Leaf Diseases Dataset},
  author={J, ARUN PANDIAN and K, GOPINATH and J, JAWAHAR},
  year={2019},
  publisher={Mendeley Data},
  doi={10.17632/tywbtsjrjv.1}
}
```

### PlantDoc Dataset
```
@article{kayal_plantdoc_2020,
  title={PlantDoc: A Dataset for Visual Plant Disease Detection},
  author={Kayal, Pratik and Singh, Shubham and Kumari, Priyanka and Kumar, Sanjeev and Balasubramanian, Venkatesh N},
  journal={Proceedings of the 35th AAAI Conference on Artificial Intelligence},
  year={2020}
}
```

### Sapling ML Project
```
@software{sapling_ml_2024,
  title={Sapling ML: Crop Disease Detection \& Recommendation System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/sapling-ml}
}
```

## Compliance and Ethics

### Data Usage Compliance
- ✅ All datasets are used in accordance with their licenses
- ✅ No copyrighted material is included without permission
- ✅ Proper attribution is provided where required
- ✅ Commercial use is permitted under the respective licenses

### Ethical Considerations
- The model is designed to assist farmers, not replace professional agronomists
- All recommendations include disclaimers about consulting professionals
- Chemical treatment recommendations require professional confirmation
- The system promotes sustainable agricultural practices

### Responsible AI
- Model predictions are explainable using Grad-CAM
- Uncertainty is communicated through confidence scores
- Recommendations prioritize cultural practices over chemical treatments
- Regular model validation ensures accuracy and reliability

## Data Quality and Validation

### Quality Assurance
- All images are validated for quality and relevance
- Duplicate images are removed using perceptual hashing
- Class labels are verified for accuracy
- Data splits are stratified to ensure balanced representation

### Validation Process
- Cross-validation on multiple datasets
- Out-of-domain testing on PlantDoc dataset
- Regular performance monitoring
- Continuous model improvement based on feedback

## Contact and Support

For questions about data usage, licensing, or compliance:
- **Email**: your.email@example.com
- **GitHub Issues**: [Project Repository](https://github.com/yourusername/sapling-ml/issues)
- **Documentation**: [Project Wiki](https://github.com/yourusername/sapling-ml/wiki)

---

*Last updated: [Current Date]*
*Version: 1.0*
