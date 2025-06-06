import React from 'react';
import PropTypes from 'prop-types';

/**
 * OnboardingTooltip component for guided tours
 * Renders a tooltip with position and styling based on the step configuration
 */
const OnboardingTooltip = (props) => {
  const { 
    title, 
    content, 
    position, 
    image, 
    currentStep, 
    totalSteps,
    onNext,
    onSkip,
    onClose
  } = props;
  
  // Position-specific arrow class
  const arrowClass = `onboarding-arrow onboarding-arrow-${position}`;
  
  return (
    <div className="onboarding-tooltip">
      <div className={arrowClass}></div>
      
      <div className="onboarding-tooltip-header">
        <h4 className="onboarding-title">{title}</h4>
        <button 
          className="onboarding-close"
          onClick={onClose}
          aria-label="Close"
        >
          ×
        </button>
      </div>
      
      <div className="onboarding-content">
        {content}
      </div>
      
      {image && (
        <div className="onboarding-image-container">
          <img 
            src={image} 
            alt={title} 
            className="onboarding-image" 
          />
        </div>
      )}
      
      <div className="onboarding-footer">
        <div className="onboarding-progress">
          <span className="onboarding-step-indicator">
            Step {currentStep} of {totalSteps}
          </span>
        </div>
        
        <div className="onboarding-buttons">
          <button 
            className="onboarding-skip"
            onClick={onSkip}
          >
            Skip
          </button>
          
          <button 
            className="onboarding-next"
            onClick={onNext}
          >
            {currentStep === totalSteps ? 'Finish' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
};

OnboardingTooltip.propTypes = {
  title: PropTypes.string.isRequired,
  content: PropTypes.node.isRequired,
  position: PropTypes.oneOf(['top', 'right', 'bottom', 'left', 'center']),
  image: PropTypes.string,
  currentStep: PropTypes.number.isRequired,
  totalSteps: PropTypes.number.isRequired,
  onNext: PropTypes.func.isRequired,
  onSkip: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired
};

OnboardingTooltip.defaultProps = {
  position: 'right',
  image: null
};

export default OnboardingTooltip;