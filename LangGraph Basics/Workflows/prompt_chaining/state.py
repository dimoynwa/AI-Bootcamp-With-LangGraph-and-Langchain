from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict, Literal

class ProductDescriptionGeneratorState(TypedDict):
    # Input
    product_id: str # ID of the product
    product_presense_status: Literal['FOUND', 'NOT_FOUND']

    # Database info
    product_name: Optional[str] # The name of the product
    product_image_url: Optional[str] # The image URL of the product
    product_features: Optional[List[str]] # Features of product
    product_category: Optional[str] # Product category
    product_specifications: Optional[Dict[str, Any]] # Technical specifications of the product

    # Output
    product_features_from_image: Optional[str] # The decription of the product from the image
    product_description: Optional[str] # The main description of the product
    product_short_description: Optional[str] # Short summary for product
    # SEO - search engine optimization
    product_seo_title: Optional[str] # SEO optimized title for product page
    product_seo_description: Optional[str] # SEO optimized meta description
    product_keywords: Optional[List[str]] # Keywords for SEO purposes