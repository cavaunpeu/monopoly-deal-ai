type Action = {
    card?: {
        name: string;
    };
    src: string;
    dst: string;
};

type Props = {
    actions: Action[];
};

export function ActionPanel({ actions }: Props) {
    return (
        <div className="border border-gray-200 rounded-lg p-4 bg-white">
            <h2 className="text-sm font-medium mb-3 text-gray-900">Available Actions</h2>
            {actions.length === 0 ? (
                <div className="text-gray-500 text-sm">No actions available</div>
            ) : (
                <div className="space-y-2">
                    {actions.map((action, i) => (
                        <div key={i} className="p-2 bg-gray-50 rounded text-sm text-gray-700">
                            {action.card?.name ? `${action.card.name} (${action.src} → ${action.dst})` : 'Pass'}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}