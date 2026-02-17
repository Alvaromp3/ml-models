import logoImage from '../../assets/logo_GUJ.jpeg';

/** Graceland Logo - Using provided image without background */
interface LogoProps {
  className?: string;
  size?: number;
}

export default function Logo({ className = '', size = 40 }: LogoProps) {
  return (
    <img
      src={logoImage}
      alt="Graceland Logo"
      className={className}
      style={{
        width: `${size}px`,
        height: `${size}px`,
        objectFit: 'contain',
        backgroundColor: 'transparent',
      }}
    />
  );
}
