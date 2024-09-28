import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function LoadingSpinner({ size = 'md', className = '' }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };

  return (
    <Loader2 className={`animate-spin ${sizeClasses[size]} ${className}`} />
  );
}

export function LoadingCard() {
  return (
    <div className="bg-white text-gray-900 rounded-lg p-3 border border-gray-200 w-[120px] h-[160px] flex flex-col justify-between animate-pulse">
      <div className="h-3 bg-gray-200 rounded"></div>
      <div className="flex-1 flex items-center justify-center">
        <div className="w-16 h-20 bg-gray-200 rounded"></div>
      </div>
      <div className="h-3 bg-gray-200 rounded"></div>
    </div>
  );
}

export function LoadingGrid({ count = 5 }: { count?: number }) {
  return (
    <div className="flex gap-2 flex-wrap">
      {Array.from({ length: count }).map((_, index) => (
        <LoadingCard key={index} />
      ))}
    </div>
  );
}
